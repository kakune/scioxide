//! Default-call equivalence and public error contracts.
#![allow(clippy::unwrap_used)]
use scioxide_core::*;
use scioxide_integrate::*;
use scioxide_random::{Philox4x64, SeedableRng};
use std::sync::atomic::{AtomicUsize, Ordering};

fn bits<S: RealScalar>(value: S) -> u64 {
    value.into_f64().to_bits()
}
fn equal<S: RealScalar>(actual: Integral<S>, expected: (S, IntegrationReport<S>)) {
    assert!(expected.1.is_success());
    assert_eq!(bits(actual.value()), bits(expected.0));
    assert_eq!(
        bits(actual.error_estimate()),
        bits(expected.1.error_estimate().unwrap())
    );
}
fn defaults<S: RealScalar + Default>() {
    let zero = S::ZERO;
    let one = S::ONE;
    let square = |x: S| Ok(x * x);
    let interval = Interval::new(zero, one).unwrap();
    let adaptive =
        AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, SubdivisionOptions::default())
            .unwrap()
            .with_extrapolation(true)
            .unwrap();
    equal(
        integrate(square, zero, one).unwrap(),
        adaptive.integrate(square, interval).unwrap(),
    );
    let points = [S::HALF];
    let piecewise =
        WeightedQuadraturePlan::breakpoints(interval, &points, Default::default()).unwrap();
    equal(
        integrate_piecewise(square, zero, one, &points).unwrap(),
        piecewise.integrate(square).unwrap(),
    );
    let weighted = WeightedQuadraturePlan::fourier(
        interval,
        &[],
        one,
        FourierWeight::Cosine,
        Default::default(),
    )
    .unwrap();
    equal(
        integrate_fourier(square, zero, one, one, FourierWeight::Cosine).unwrap(),
        weighted.integrate(square).unwrap(),
    );
    let decay = |x: S| Ok((-x).exp());
    let half = DoubleExponentialIntegrator::new(
        DoubleExponentialDomain::HalfLine { lower: zero },
        LevelOptions::default(),
    )
    .unwrap();
    equal(
        integrate_unbounded(decay, ExtendedInterval::new(Some(zero), None).unwrap()).unwrap(),
        half.integrate(|p: DoubleExponentialPoint<S>| decay(p.coordinate()))
            .unwrap(),
    );
    let reflected = DoubleExponentialIntegrator::new(
        DoubleExponentialDomain::HalfLine { lower: -one },
        LevelOptions::default(),
    )
    .unwrap();
    equal(
        integrate_unbounded(
            |x: S| Ok((x - one).exp()),
            ExtendedInterval::new(None, Some(one)).unwrap(),
        )
        .unwrap(),
        reflected
            .integrate(|p: DoubleExponentialPoint<S>| Ok((-p.coordinate() - one).exp()))
            .unwrap(),
    );
    let cube = CubaturePlan::new(&[interval; 2], SubdivisionOptions::default()).unwrap();
    let product = |x: &[S]| Ok(x[0] * x[1]);
    equal(
        integrate_box(product, &[zero; 2], &[one; 2]).unwrap(),
        cube.integrate(product).unwrap(),
    );
    let bounds = |outer: &[S]| Interval::new(zero, outer.first().copied().unwrap_or(one));
    let nested = IteratedDomainPlan::new(2, IteratedOptions::default()).unwrap();
    equal(
        integrate_iterated(product, bounds, 2).unwrap(),
        nested.integrate(product, bounds).unwrap(),
    );
    let coordinates = [zero, S::HALF, one];
    let samples = coordinates.map(|x| x * x);
    let trap = SampledTrapezoidPlan::new(&coordinates).unwrap();
    let simp = SampledSimpsonPlan::new(&coordinates).unwrap();
    let mut workspace = trap.cumulative_workspace(1).unwrap();
    let mut expected = zero;
    trap.integrate_into(&samples, &mut expected, &mut workspace)
        .unwrap();
    assert_eq!(
        bits(trapezoid(&samples, &coordinates).unwrap()),
        bits(expected)
    );
    simp.integrate_into(&samples, &mut expected, &mut workspace)
        .unwrap();
    assert_eq!(
        bits(simpson(&samples, &coordinates).unwrap()),
        bits(expected)
    );
    let mut prefix = [zero; 3];
    trap.cumulative_into(&samples, &mut prefix, &mut workspace)
        .unwrap();
    let actual = cumulative_trapezoid(&samples, &coordinates).unwrap();
    for (a, b) in actual.iter().zip(prefix) {
        assert_eq!(bits(*a), bits(b));
    }
    let term = |n| Ok(S::HALF.powi(n as i32));
    let sum = sum_series(term, 0).unwrap();
    let plan = SeriesSummation::new(SeriesMethod::LevinU, 0, SeriesOptions::default()).unwrap();
    let expected = plan
        .sum(SeriesProblem::Terms(&|n, row: &mut [S]| {
            row[0] = term(n)?;
            Ok(())
        }))
        .unwrap();
    assert_eq!(bits(sum.value()), bits(expected.value()));
    assert_eq!(sum.report().termination(), expected.report().termination());
    let rng = Philox4x64::from_seed(7);
    let f = |x: &[S]| Ok(x[0] * x[0]);
    let actual = integrate_monte_carlo(f, &[zero], &[one], &rng).unwrap();
    let repeated = integrate_monte_carlo(f, &[zero], &[one], &rng).unwrap();
    let plan = MonteCarloIntegrator::new(&[interval], MonteCarloOptions::default()).unwrap();
    let context = ExecutionContext::sequential_native();
    let mut workspace = plan.workspace(1, context).unwrap();
    let mut value = zero;
    let report = plan
        .integrate_into(
            f,
            RandomStreams::new(&rng, 0),
            &mut value,
            &mut workspace,
            context,
        )
        .unwrap();
    assert!(report.is_success());
    let configured = workspace.statistics()[0].as_ref().unwrap();
    assert_eq!(bits(actual.estimate()), bits(configured.estimate()));
    assert_eq!(
        bits(actual.standard_error()),
        bits(configured.standard_error())
    );
    assert_eq!(actual.samples(), 4096);
    assert_eq!(bits(actual.estimate()), bits(repeated.estimate()));
    let other = integrate_monte_carlo(f, &[zero], &[one], &Philox4x64::from_seed(8)).unwrap();
    assert_ne!(bits(actual.estimate()), bits(other.estimate()));
}

#[test]
fn all_default_calls_preserve_scalar_bits() {
    defaults::<f64>();
    defaults::<f32>();
}

#[test]
fn endpoint_singularities_use_extrapolation() {
    let configured =
        AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, SubdivisionOptions::default())
            .unwrap()
            .with_extrapolation(true)
            .unwrap();
    let interval = Interval::new(0., 1.).unwrap();
    for (f, expected, tolerance) in [
        (
            (|x: f64| Ok(1. / x.sqrt())) as fn(f64) -> Result<f64, SciOxideError>,
            2.,
            1e-7,
        ),
        (|x: f64| Ok(x.powf(-0.9)), 10., 1e-7),
        (|x: f64| Ok(x.ln()), -1., 1e-9),
    ] {
        let actual = integrate(f, 0., 1.).unwrap();
        assert!((actual.value() - expected).abs() < tolerance);
        equal(actual, configured.integrate(f, interval).unwrap());
    }
    let ordinary =
        AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, SubdivisionOptions::default()).unwrap();
    let (_, report) = ordinary
        .integrate(|x: f64| Ok(x.powf(-0.9)), interval)
        .unwrap();
    assert_eq!(
        report.termination(),
        IntegrationTermination::BudgetLimit(BudgetKind::Subdivision)
    );
}

#[test]
fn unbounded_domains_and_plain_callbacks() {
    let upper = integrate_unbounded(
        |x: f64| Ok((-x).exp()),
        ExtendedInterval::new(Some(0.), None).unwrap(),
    )
    .unwrap();
    let lower = integrate_unbounded(
        |x: f64| Ok(x.exp()),
        ExtendedInterval::new(None, Some(0.)).unwrap(),
    )
    .unwrap();
    let line =
        integrate_unbounded(|x: f64| Ok((-x * x).exp()), ExtendedInterval::entire()).unwrap();
    assert!((upper.value() - 1.).abs() < 1e-8);
    assert!((lower.value() - 1.).abs() < 1e-8);
    assert!((line.value() - core::f64::consts::PI.sqrt()).abs() < 1e-8);
    let calls = AtomicUsize::new(0);
    for (domain, reason) in [
        (ExtendedInterval::empty(), InvalidInputReason::Empty),
        (
            ExtendedInterval::new(Some(0.), Some(1.)).unwrap(),
            InvalidInputReason::Inconsistent,
        ),
    ] {
        let e = integrate_unbounded(
            |_: f64| {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(1.)
            },
            domain,
        )
        .unwrap_err();
        assert!(
            matches!(e.view(),SciOxideErrorView::InvalidInput {argument,reason:r,..} if argument.as_str()=="domain" && r==reason)
        );
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn fixed_arrays_complex_values_and_exact_degeneracies() {
    let pair = integrate(|x: f64| Ok([x, x * x]), 0., 1.).unwrap();
    assert!((pair.value()[0] - 0.5).abs() < 1e-14);
    assert!((pair.value()[1] - 1. / 3.).abs() < 1e-14);
    let complex = integrate(|x: f64| Ok(Complex::new(x, x * x)), 0., 1.).unwrap();
    assert!((complex.value().re - 0.5).abs() < 1e-14);
    assert!((complex.value().im - 1. / 3.).abs() < 1e-14);
    let calls = AtomicUsize::new(0);
    let scalar = |_: f64| {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok(1.)
    };
    for result in [
        integrate(scalar, 1., 1.),
        integrate_piecewise(scalar, 1., 1., &[]),
        integrate_fourier(scalar, 1., 1., 1., FourierWeight::Cosine),
        integrate_box(
            |_: &[f64]| {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(1.)
            },
            &[0., 1.],
            &[1., 1.],
        ),
    ] {
        let result = result.unwrap();
        assert_eq!(result.value(), 0.);
        assert_eq!(result.error_estimate(), 0.);
    }
    let empty = integrate(
        |_: f64| {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok([] as [f64; 0])
        },
        0.,
        1.,
    )
    .unwrap();
    assert_eq!(empty.value(), []);
    assert_eq!(empty.error_estimate(), 0.);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let point = integrate_box(
        |x: &[f64]| {
            assert!(x.is_empty());
            Ok(3.)
        },
        &[],
        &[],
    )
    .unwrap();
    assert_eq!(point.value(), 3.);
    assert_eq!(point.error_estimate(), 0.);
}

#[test]
fn errors_name_the_input_and_keep_callback_failures() {
    for bound in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        for error in [
            integrate(Ok, bound, 1.).unwrap_err(),
            integrate(Ok, 0., bound).unwrap_err(),
        ] {
            assert!(matches!(
                error.view(),
                SciOxideErrorView::NonFinite {
                    origin: NonFiniteOrigin::Input,
                    ..
                }
            ));
        }
    }
    let error = integrate(Ok, 1., 0.).unwrap_err();
    assert!(
        matches!(error.view(),SciOxideErrorView::InvalidInput {argument,..} if argument.as_str()=="lower")
    );
    let callback = SciOxideError::not_implemented("caller failure");
    assert_eq!(
        integrate(|_: f64| Err::<f64, _>(callback.clone()), 0., 1.).unwrap_err(),
        callback
    );
    assert!(matches!(
        integrate(|_: f64| Ok(f64::NAN), 0., 1.).unwrap_err().view(),
        SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Callback,
            ..
        }
    ));
    for (error, reason) in [
        (
            integrate(|x: f64| Ok(1. / x), 0., 1.).unwrap_err(),
            ConvergenceReason::Divergence,
        ),
        (
            integrate_unbounded(
                |_: f64| Ok(1.),
                ExtendedInterval::new(Some(0.), None).unwrap(),
            )
            .unwrap_err(),
            ConvergenceReason::Stagnation,
        ),
        (
            sum_series(|n| Ok(1.0_f64 / (n + 1) as f64), 0).unwrap_err(),
            ConvergenceReason::TruncationLimit,
        ),
    ] {
        assert!(
            matches!(error.view(),SciOxideErrorView::ConvergenceFailure {reason:r,..} if r==reason)
        );
    }
    let calls = AtomicUsize::new(0);
    let root = Philox4x64::from_seed(1);
    let integrand = |_: &[f64]| {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok(1.)
    };
    assert_eq!(
        integrate_box(integrand, &[0.], &[]).unwrap_err().kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        integrate_monte_carlo(integrand, &[0.], &[], &root)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    for (lo, hi, name) in [(&[][..], &[][..], "lower"), (&[0.][..], &[0.][..], "upper")] {
        assert!(
            matches!(integrate_monte_carlo(integrand,lo,hi,&root).unwrap_err().view(),SciOxideErrorView::InvalidInput {argument,..} if argument.as_str()==name)
        );
    }
    let d = usize::BITS as usize;
    assert!(
        matches!(integrate_box(integrand,&vec![0.;d],&vec![1.;d]).unwrap_err().view(),SciOxideErrorView::InvalidInput {argument,..} if argument.as_str()=="lower")
    );
    assert!(
        matches!(sum_series(|_|{calls.fetch_add(1,Ordering::Relaxed);Ok(1.)},usize::MAX).unwrap_err().view(),SciOxideErrorView::InvalidInput {argument,..} if argument.as_str()=="first")
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(
        sum_series(|_| Err::<f64, _>(callback.clone()), 0).unwrap_err(),
        callback
    );
}

#[test]
fn sampled_edges_and_infinite_series_values() {
    for (samples, coords) in [(&[][..], &[][..]), (&[12.][..], &[1.][..])] {
        assert_eq!(trapezoid(samples, coords).unwrap(), 0.);
        assert_eq!(simpson(samples, coords).unwrap(), 0.);
        assert!(
            cumulative_trapezoid(samples, coords)
                .unwrap()
                .iter()
                .all(|&x| x == 0.)
        );
    }
    let coords = [1., 0.5, 0.];
    let samples = [1., 0.25, 0.];
    assert!((simpson(&samples, &coords).unwrap() + 1. / 3.).abs() < 1e-14);
    assert_eq!(
        cumulative_trapezoid(&samples, &coords).unwrap()[2],
        trapezoid(&samples, &coords).unwrap()
    );
    assert_eq!(
        trapezoid(&[1., 2.], &[0., 0.]).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        simpson(&[1., 2.], &[0.]).unwrap_err().kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    let complex = [Complex::new(0., 1.), Complex::new(1., 2.)];
    let result = trapezoid(&complex, &[0., 1.]).unwrap();
    assert!((result.re - 0.5).abs() < 1e-14);
    assert!((result.im - 1.5).abs() < 1e-14);
    assert!(
        (sum_series(|n| Ok(0.5_f64.powi(n as i32)), 0)
            .unwrap()
            .value()
            - 2.)
            .abs()
            < 1e-6
    );
    assert!(
        (sum_series(|n| Ok(1.0_f64 / (n as f64).powi(2)), 1)
            .unwrap()
            .value()
            - core::f64::consts::PI.powi(2) / 6.)
            .abs()
            < 1e-6
    );
}
