//! Independent rational-function identities and public preparation workflows.
#![allow(clippy::unwrap_used)]
mod support;

use support::assert_error;

use scioxide_core::{
    Complex, ConvergenceReason, Interval, InvalidInputReason, NumericScalar, RealScalar,
    SciOxideErrorKind, SciOxideErrorView, Tolerance,
};
use scioxide_interp::{
    AaaOptions, BarycentricRational, ExtrapolationPolicy as E, FloaterHormannInterpolator,
    Interpolant, PadeApproximant, RationalDomain, RationalFit, RationalTermination, aaa,
    aaa_with_options,
};
use scioxide_linalg::RankPolicy;
use scioxide_poly::PolynomialApproximationPolicy;
fn domain() -> RationalDomain<f64> {
    RationalDomain::Real(Interval::new(-1., 1.).unwrap())
}
fn options<R: RealScalar>(max_terms: usize, tolerance: Tolerance<R>) -> AaaOptions<R> {
    AaaOptions::new()
        .with_max_terms(max_terms)
        .with_tolerance(tolerance)
}
fn termination<C: scioxide_core::NumericScalar>(fit: &RationalFit<C>) -> RationalTermination {
    fit.report().unwrap().termination()
}
fn max_residual<C: scioxide_core::NumericScalar>(fit: &RationalFit<C>) -> C::Real {
    fit.report().unwrap().max_residual()
}
fn at(fit: &RationalFit<f64>, z: f64) -> f64 {
    fit.approximant().unwrap().evaluate([z], E::Reject).unwrap()
}
#[test]
fn aaa_recovers_reciprocal() {
    let x: Vec<_> = (0..21).map(|i| -1. + i as f64 / 10.).collect();
    let y: Vec<_> = x.iter().map(|x| 1. / (2. + x)).collect();
    let result = aaa_with_options(
        &x,
        &y,
        domain(),
        options(6, Tolerance::new(1e-12, 1e-12).unwrap()),
    )
    .unwrap();
    assert_eq!(termination(&result), RationalTermination::Converged);
    for i in 0..100 {
        let z = -1. + i as f64 / 50.;
        assert!((at(&result, z) - 1. / (2. + z)).abs() < 1e-12);
    }
}
#[test]
fn floater_hormann_reproduces_quadratic() {
    let x = [-1., -0.7, -0.1, 0.4, 1.];
    let y = x.map(|x| x * x + 2. * x);
    let plan = FloaterHormannInterpolator::new(x.to_vec(), y.to_vec(), 2).unwrap();
    for i in 0..100 {
        let z = -1. + i as f64 / 50.;
        assert!((plan.evaluate([z], E::Reject).unwrap() - (z * z + 2. * z)).abs() < 1e-13);
    }
}
#[test]
fn weights_are_scale_invariant() {
    for scale in [1., 1e-200, 1e200] {
        let p = BarycentricRational::<f64>::new(
            vec![-1., 1.],
            vec![scale, scale],
            vec![scale, 2. * scale],
            domain(),
        )
        .unwrap();
        assert!((p.evaluate([0.], E::Reject).unwrap()).abs() < 1e-14);
        assert!((p.evaluate([-1.], E::Reject).unwrap() - 1.).abs() < 1e-14);
    }
}
#[test]
fn pade_exp_22() {
    let policy = PolynomialApproximationPolicy::new(
        Tolerance::new(1e-13, 1e-13).unwrap(),
        RankPolicy::DimensionScaledEpsilon,
    );
    let p = PadeApproximant::<f64>::with_options(
        &[1., 1., 0.5, 1. / 6., 1. / 24.],
        2,
        2,
        0.,
        domain(),
        policy,
    )
    .unwrap();
    for z in [-0.5, 0., 0.5] {
        let expected = (1. + z / 2. + z * z / 12.) / (1. - z / 2. + z * z / 12.);
        assert!((p.evaluate([z], E::Reject).unwrap() - expected).abs() < 1e-13);
    }
}
#[test]
fn complex_aaa() {
    let d = RationalDomain::Complex {
        real: Interval::new(-1., 1.).unwrap(),
        imaginary: Interval::new(-1., 1.).unwrap(),
    };
    let x: Vec<_> = (0..16)
        .map(|i| {
            let t = i as f64 * core::f64::consts::TAU / 16.;
            Complex::new(t.cos(), t.sin())
        })
        .collect();
    let y: Vec<_> = x
        .iter()
        .map(|&z| Complex::new(1., 0.) / (z + Complex::new(2., 0.)))
        .collect();
    let fit =
        aaa_with_options(&x, &y, d, options(5, Tolerance::new(1e-12, 1e-12).unwrap())).unwrap();
    assert_eq!(termination(&fit), RationalTermination::Converged);
    assert!(
        (fit.approximant()
            .unwrap()
            .evaluate([0., 0.3], E::Reject)
            .unwrap()
            - Complex::new(1., 0.) / Complex::new(2., 0.3))
        .norm()
            < 1e-12
    );
}

#[test]
fn poles_residues_and_explicit_doublet_cleanup() {
    let support = [-1.0_f64, 0., 1.];
    let numerator: Vec<_> = support
        .iter()
        .enumerate()
        .map(|(j, &s)| {
            (s - 0.2)
                / support
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| *k != j)
                    .map(|(_, &t)| s - t)
                    .product::<f64>()
        })
        .collect();
    let denominator: Vec<_> = support
        .iter()
        .enumerate()
        .map(|(j, &s)| {
            (s - 2.) * (s - 0.20000001)
                / support
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| *k != j)
                    .map(|(_, &t)| s - t)
                    .product::<f64>()
        })
        .collect();
    let plan =
        BarycentricRational::new(support.to_vec(), numerator, denominator, domain()).unwrap();
    let policy = PolynomialApproximationPolicy::new(
        Tolerance::new(1e-13, 1e-13).unwrap(),
        RankPolicy::DimensionScaledEpsilon,
    );
    let report = plan
        .diagnostics(
            policy,
            Tolerance::new(1e-6, 1e-6).unwrap(),
            &Default::default(),
        )
        .unwrap();
    assert_eq!(report.poles().len(), 2);
    assert_eq!(report.roots().len(), 1);
    let p = report
        .poles()
        .iter()
        .find(|p| p.location().re > 1.)
        .unwrap();
    assert!((p.location().re - 2.).abs() < 1e-12);
    assert!((p.residue().unwrap().re - 1.).abs() < 1e-7);
    assert_eq!(
        report
            .poles()
            .iter()
            .filter(|p| p.suspected_doublet())
            .count(),
        1
    );
    let x = [-1., -0.5, 0., 0.5, 1.];
    let y = x.map(|s| (s - 0.2) / ((s - 2.) * (s - 0.20000001)));
    let cleanup = plan
        .cleanup_doublets(
            &x,
            &y,
            policy,
            Tolerance::new(1e-6, 1e-6).unwrap(),
            &Default::default(),
        )
        .unwrap();
    assert_eq!(cleanup.removed_supports(), 1);
    assert!(cleanup.before_residual() < 1e-12);
    assert!(cleanup.after_residual() < 1e-6);
    assert_eq!(cleanup.approximant().support_points().len(), 2);
    assert_eq!(plan.support_points().len(), 3);
}
#[test]
fn full_support_and_constant_samples() {
    for y in [[1., 2., -1.], [3., 3., 3.], [0., 0., 0.]] {
        let fit = aaa_with_options(
            &[-1.0_f64, 0., 1.],
            &y,
            domain(),
            options(3, Tolerance::new(1e-14, 1e-14).unwrap()),
        )
        .unwrap();
        assert_eq!(termination(&fit), RationalTermination::Converged);
        for (z, f) in [-1., 0., 1.].into_iter().zip(y) {
            assert!((at(&fit, z) - f).abs() < 1e-13);
        }
    }
}
#[test]
fn lawson_improves_a_fixed_degree_fit() {
    let x: Vec<_> = (0..101).map(|i| -1. + i as f64 / 50.).collect();
    let y: Vec<_> = x.iter().map(|x| x.exp()).collect();
    let options = options(3, Tolerance::new(1e-13, 1e-13).unwrap());
    let plain = aaa_with_options(&x, &y, domain(), options).unwrap();
    let lawson = aaa_with_options(&x, &y, domain(), options.with_lawson_iterations(20)).unwrap();
    assert!(
        max_residual(&lawson) < max_residual(&plain) * 0.9,
        "AAA {} Lawson {}",
        max_residual(&plain),
        max_residual(&lawson)
    );
    let mut offgrid = 0.0_f64;
    for i in 0..100 {
        let z = -0.99 + i as f64 / 50.;
        offgrid = offgrid.max((at(&lawson, z) - z.exp()).abs());
    }
    assert!(offgrid < 1e-3);
}
#[test]
fn both_scalar_widths_and_complex_values() {
    let x = [-1.0_f32, -0.5, 0., 0.5, 1.];
    let y = x.map(|x| 1. / (2. + x));
    let d = RationalDomain::Real(Interval::new(-1.0_f32, 1.).unwrap());
    let fit = aaa_with_options(&x, &y, d, options(3, Tolerance::new(2e-6, 2e-6).unwrap())).unwrap();
    assert_eq!(termination(&fit), RationalTermination::Converged);
    assert!(
        (fit.approximant()
            .unwrap()
            .evaluate([0.25], E::Reject)
            .unwrap()
            - 1. / 2.25)
            .abs()
            < 2e-6
    );
    let y = x.map(|x| Complex::new(x * x, x));
    let fh = FloaterHormannInterpolator::new(x.to_vec(), y.to_vec(), 2).unwrap();
    assert!((fh.evaluate([0.25], E::Reject).unwrap() - Complex::new(0.0625, 0.25)).norm() < 2e-6);
}

#[test]
fn poles_domains_views_and_invalid_inputs() {
    let pole =
        BarycentricRational::<f64>::new(vec![-1., 1.], vec![1., -1.], vec![1., 1.], domain())
            .unwrap();
    assert_error(
        &pole.evaluate([0.], E::Reject),
        scioxide_core::SciOxideErrorKind::NumericalInstability,
    );
    let near = pole.evaluate([1e-7], E::Reject).unwrap();
    assert!((near * 1e-7 + 1.).abs() < 1e-8);
    assert!(pole.denominator_condition(1e-7, E::Reject).unwrap() > 1e6);
    assert_error(
        &pole.evaluate([2.], E::Reject),
        scioxide_core::SciOxideErrorKind::DomainError,
    );
    assert_eq!(pole.evaluate([2.], E::Clamp).unwrap(), -1.);
    assert!(
        pole.evaluate([1.5], E::BoundedContinue { max_distance: 0.5 })
            .is_ok()
    );
    assert!(
        pole.evaluate([1.6], E::BoundedContinue { max_distance: 0.5 })
            .is_err()
    );
    assert_eq!(pole.evaluate([1.], E::Reject).unwrap(), -1.);
    let values = pole
        .support_points()
        .iter()
        .map(|&z| pole.evaluate([z], E::Reject).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(values, [1., -1.]);
    assert!(
        BarycentricRational::<f64>::new(vec![0., 0.], vec![1., 1.], vec![1., 1.], domain())
            .is_err()
    );
    assert!(BarycentricRational::<f64>::new(vec![0.], vec![1.], vec![0.], domain()).is_err());
    assert!(FloaterHormannInterpolator::new(vec![0.0_f64, 0.], vec![1.0_f64, 2.], 1).is_err());
    assert!(
        aaa_with_options(
            &[0.0_f64, 1.],
            &[1.0_f64, f64::NAN],
            domain(),
            options(2, Tolerance::new(1e-12, 1e-12).unwrap())
        )
        .is_err()
    );
}
#[test]
fn full_range_query_arithmetic() {
    let d = RationalDomain::Real(Interval::new(-f64::MAX, f64::MAX).unwrap());
    let p =
        BarycentricRational::<f64>::new(vec![-f64::MAX, f64::MAX], vec![1., 1.], vec![1., 1.], d)
            .unwrap();
    assert_eq!(p.evaluate([f64::MAX / 2.], E::Reject).unwrap(), 1.);
    let tiny = f64::from_bits(1);
    for weight in [tiny, 1., 1e300] {
        let p = BarycentricRational::<f64>::new(
            vec![-1., 1.],
            vec![weight, weight],
            vec![weight, weight],
            domain(),
        )
        .unwrap();
        assert_eq!(p.evaluate([0.5], E::Reject).unwrap(), 1.);
    }
    let p = BarycentricRational::<f64>::new(
        vec![0., tiny],
        vec![tiny, tiny],
        vec![1., 1.],
        RationalDomain::Real(Interval::new(0., 1.).unwrap()),
    )
    .unwrap();
    assert_eq!(p.evaluate([0.5], E::Reject).unwrap(), tiny);
}
#[test]
fn pade_defects_and_f32() {
    let policy = PolynomialApproximationPolicy::new(
        Tolerance::new(2e-6_f32, 2e-6).unwrap(),
        RankPolicy::DimensionScaledEpsilon,
    );
    let d = RationalDomain::Real(Interval::new(-1.0_f32, 1.).unwrap());
    let p =
        PadeApproximant::<f32>::with_options(&[1., 1., 1., 1., 1.], 2, 2, 0., d, policy).unwrap();
    assert_eq!(p.denominator_rank(), 1);
    assert!((p.evaluate([0.2], E::Reject).unwrap() - 1.25).abs() < 2e-6);
    assert!(PadeApproximant::<f32>::with_options(&[0., 0., 1.], 1, 1, 0., d, policy).is_err());
}

#[test]
fn independent_exact_fraction_and_high_precision_fixtures() {
    let x64: Vec<_> = (-4..=4).map(|i| i as f64 / 4.).collect();
    let y64: Vec<_> = x64.iter().map(|x| 1. / (1. + 25. * x * x)).collect();
    let x32: Vec<_> = x64.iter().map(|&x| x as f32).collect();
    let y32: Vec<_> = y64.iter().map(|&y| y as f32).collect();
    let fh64 = FloaterHormannInterpolator::new(x64, y64, 3).unwrap();
    let fh32 = FloaterHormannInterpolator::new(x32, y32, 3).unwrap();
    let samples: Vec<_> = (0..33).map(|i| -1. + i as f64 / 16.).collect();
    let values: Vec<_> = samples.iter().map(|x| x.exp()).collect();
    let fit = aaa_with_options(
        &samples,
        &values,
        domain(),
        options(6, Tolerance::new(1e-11, 1e-11).unwrap()),
    )
    .unwrap();
    for line in include_str!("data/rational.csv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let row: Vec<_> = line.split(',').collect();
        let z: f64 = row[2].parse().unwrap();
        let expected: f64 = row[3].parse().unwrap();
        let (actual, tol) = match (row[0], row[1]) {
            ("fh", "32") => (fh32.evaluate([z as f32], E::Reject).unwrap() as f64, 3e-6),
            ("fh", "64") => (fh64.evaluate([z], E::Reject).unwrap(), 2e-13),
            _ => (at(&fit, z), 2e-10),
        };
        assert!((actual - expected).abs() < tol, "{line}: got {actual}");
    }
}
#[test]
fn noisy_samples_have_an_explicit_residual_floor() {
    let x: Vec<_> = (0..65).map(|i| -1. + i as f64 / 32.).collect();
    let y: Vec<_> = x
        .iter()
        .enumerate()
        .map(|(i, x)| 1. / (2. + x) + if i % 2 == 0 { 1e-5 } else { -1e-5 })
        .collect();
    let fit = aaa_with_options(
        &x,
        &y,
        domain(),
        options(8, Tolerance::new(4e-5, 0.).unwrap()),
    )
    .unwrap();
    assert_eq!(termination(&fit), RationalTermination::Converged);
    assert!(max_residual(&fit) <= 4e-5);
    for i in 0..32 {
        let z = -0.984375 + i as f64 / 16.;
        assert!((at(&fit, z) - 1. / (2. + z)).abs() < 1e-4);
    }
}

#[test]
fn complex_pade_and_complex32_aaa() {
    let d = RationalDomain::Complex {
        real: Interval::new(-1.0_f32, 1.).unwrap(),
        imaginary: Interval::new(-1., 1.).unwrap(),
    };
    let z = [
        Complex::new(-1.0_f32, 0.),
        Complex::new(0., -1.),
        Complex::new(1., 0.),
        Complex::new(0., 1.),
        Complex::new(0., 0.),
    ];
    let y = z.map(|z| Complex::new(1., 1.) / (z + Complex::new(2., 0.)));
    let fit = aaa_with_options(&z, &y, d, options(3, Tolerance::new(3e-6, 3e-6).unwrap())).unwrap();
    let query = Complex::new(0.3, 0.2);
    assert!(
        (fit.approximant()
            .unwrap()
            .evaluate([query.re, query.im], E::Reject)
            .unwrap()
            - Complex::new(1., 1.) / (query + Complex::new(2., 0.)))
        .norm()
            < 1e-5
    );
    let coefficients = [
        Complex::new(1., 1.),
        Complex::new(-0.5, -0.5),
        Complex::new(0.25, 0.25),
    ];
    let policy = PolynomialApproximationPolicy::new(
        Tolerance::new(3e-6, 3e-6).unwrap(),
        RankPolicy::DimensionScaledEpsilon,
    );
    let p = PadeApproximant::with_options(&coefficients, 1, 1, Complex::new(0., 0.), d, policy)
        .unwrap();
    assert!(
        (p.evaluate([query.re, query.im], E::Reject).unwrap()
            - Complex::new(2., 2.) / (query + Complex::new(2., 0.)))
        .norm()
            < 1e-5
    );
}

#[test]
fn diagnostic_trimming_cannot_publish_a_zero_denominator() {
    let plan =
        BarycentricRational::<f64>::new(vec![0., 1.], vec![1., 1.], vec![0., 1e-12], domain())
            .unwrap();
    let policy = PolynomialApproximationPolicy::new(
        Tolerance::new(1e-6, 0.).unwrap(),
        RankPolicy::DimensionScaledEpsilon,
    );
    assert_error(
        &plan.diagnostics(
            policy,
            Tolerance::new(1e-6, 0.).unwrap(),
            &Default::default(),
        ),
        scioxide_core::SciOxideErrorKind::NumericalInstability,
    );
}

#[test]
fn lawson_complex_coordinates_have_independent_off_grid_accuracy() {
    let d = RationalDomain::Complex {
        real: Interval::new(-1.0_f64, 1.).unwrap(),
        imaginary: Interval::new(-1., 1.).unwrap(),
    };
    let exponential =
        |z: Complex<f64>| Complex::new(z.re.exp() * z.im.cos(), z.re.exp() * z.im.sin());
    let nodes: Vec<_> = (0..25)
        .map(|i| {
            let t = f64::from(i) * core::f64::consts::TAU / 25.;
            Complex::new(t.cos(), t.sin())
        })
        .collect();
    let values: Vec<_> = nodes.iter().copied().map(exponential).collect();
    let options = options(3, Tolerance::new(1e-12, 1e-12).unwrap());
    let plain = aaa_with_options(&nodes, &values, d, options).unwrap();
    let refined = aaa_with_options(&nodes, &values, d, options.with_lawson_iterations(10)).unwrap();
    assert!(max_residual(&refined) <= max_residual(&plain));
    let query = Complex::new(0.2, -0.2);
    let value = refined
        .approximant()
        .unwrap()
        .evaluate([query.re, query.im], E::Reject)
        .unwrap();
    assert!((value - exponential(query)).norm() < 0.002);
}

#[test]
fn floater_hormann_normalizes_full_range_coordinates() {
    let nodes = [-1.0_f64, -0.5, 0., 0.5, 1.];
    let values = nodes.map(|x| x * x);
    let wide = nodes.map(|x| x * f64::MAX);
    let plan = FloaterHormannInterpolator::new(wide.to_vec(), values.to_vec(), 2).unwrap();
    assert!((plan.evaluate([f64::MAX / 4.], E::Reject).unwrap() - 0.0625).abs() < 1e-13);
}

/// Compares two approximants bit for bit.
fn assert_same<C: NumericScalar>(left: &BarycentricRational<C>, right: &BarycentricRational<C>) {
    assert_eq!(left.domain(), right.domain());
    let bits = |values: &[C]| {
        values
            .iter()
            .flat_map(|&value| [value.real_component(), value.imaginary_component()])
            .map(|part| part.try_to_f64_exact().unwrap().to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(left.support_points()), bits(right.support_points()));
    assert_eq!(
        bits(left.numerator_weights()),
        bits(right.numerator_weights())
    );
    assert_eq!(bits(left.weights()), bits(right.weights()));
}

/// Runs the configured form with the defaults of the one-shot form.
fn configured<C: NumericScalar>(
    nodes: &[C],
    values: &[C],
    domain: RationalDomain<C::Real>,
) -> BarycentricRational<C> {
    let options = AaaOptions::default().with_max_terms(100.min(nodes.len()));
    let fit = aaa_with_options(nodes, values, domain, options).unwrap();
    assert_eq!(termination(&fit), RationalTermination::Converged);
    fit.into_approximant().unwrap()
}

#[test]
fn one_shot_aaa_equals_the_configured_form_on_the_node_box() {
    let x: Vec<f64> = (0..41).map(|i| -0.5 + f64::from(i) / 20.).collect();
    let y: Vec<f64> = x.iter().map(|x| x.exp() / (3. + x)).collect();
    let real = RationalDomain::Real(Interval::new(-0.5, 1.5).unwrap());
    let r = aaa(&x, &y).unwrap();
    assert_same(&r, &configured(&x, &y, real));
    assert!((r.evaluate([0.3], E::Reject).unwrap() - 0.3_f64.exp() / 3.3).abs() < 1e-7);
    // Three samples: the limit is the sample count.
    let r = aaa(&[0.0_f64, 1., 2.], &[1., 0., 1.]).unwrap();
    assert_same(&r, &configured(&[0., 1., 2.], &[1., 0., 1.], r.domain()));

    let x32: Vec<f32> = x.iter().map(|&x| x as f32).collect();
    let y32: Vec<f32> = x32.iter().map(|x| 1. / (3. + x)).collect();
    let r = aaa(&x32, &y32).unwrap();
    let real32 = RationalDomain::Real(Interval::new(-0.5_f32, 1.5).unwrap());
    assert_same(&r, &configured(&x32, &y32, real32));
    assert!((r.evaluate([0.3], E::Reject).unwrap() - 1. / 3.3).abs() < 1e-5);
}

#[test]
fn one_shot_aaa_derives_real_and_rectangle_domains_of_complex_nodes() {
    let reciprocal = |z: Complex<f64>| Complex::new(1., 0.) / (z + Complex::new(3., 0.));
    // Real nodes in a complex field give a real interval.
    let axis: Vec<_> = (0..9)
        .map(|i| Complex::new(-1. + f64::from(i) / 4., 0.))
        .collect();
    let values: Vec<_> = axis.iter().copied().map(reciprocal).collect();
    let r = aaa(&axis, &values).unwrap();
    assert_eq!(
        r.domain(),
        RationalDomain::Real(Interval::new(-1., 1.).unwrap())
    );
    assert_same(&r, &configured(&axis, &values, r.domain()));
    // Off-axis nodes give the rectangle of their components.
    let plane = [
        (-1., 0.),
        (0., -0.5),
        (1., 0.),
        (0., 0.5),
        (0.25, 0.25),
        (-0.5, -0.25),
    ]
    .map(|(re, im)| Complex::new(re, im));
    let values = plane.map(reciprocal);
    let r = aaa(&plane, &values).unwrap();
    let rectangle = RationalDomain::Complex {
        real: Interval::new(-1., 1.).unwrap(),
        imaginary: Interval::new(-0.5, 0.5).unwrap(),
    };
    assert_eq!(r.domain(), rectangle);
    assert_same(&r, &configured(&plane, &values, rectangle));
    let query = Complex::new(0.3, 0.2);
    let value = Interpolant::<f64, 2>::evaluate(&r, [query.re, query.im], E::Reject).unwrap();
    assert!((value - reciprocal(query)).norm() < 1e-12);

    let plane32 = plane.map(|z| Complex::new(z.re as f32, z.im as f32));
    let values32 = plane32.map(|z| Complex::new(1.0_f32, 0.) / (z + Complex::new(3., 0.)));
    let r = aaa(&plane32, &values32).unwrap();
    assert!(matches!(r.domain(), RationalDomain::Complex { .. }));
    assert_same(&r, &configured(&plane32, &values32, r.domain()));
}

#[test]
fn one_shot_aaa_reports_the_support_limit_as_truncation() {
    let x: Vec<f64> = (0..500).map(|i| -1. + f64::from(i) / 249.5).collect();
    let y: Vec<f64> = (1..=500)
        .map(|k| (0.6180339887 * f64::from(k)).fract())
        .collect();
    let error = aaa(&x, &y).unwrap_err();
    assert!(
        matches!(
            error.view(),
            SciOxideErrorView::ConvergenceFailure {
                reason: ConvergenceReason::TruncationLimit,
                diagnostics,
                ..
            } if diagnostics.algorithm() == Some(scioxide_core::algorithm_id!("interp", "aaa", 1))
                && diagnostics.counts().own().iterations() == 100
        ),
        "{error:?}"
    );
}

#[test]
fn one_shot_aaa_rejects_empty_mismatched_and_invalid_samples() {
    let empty = aaa::<f64>(&[], &[]);
    assert!(matches!(
        assert_error(&empty, SciOxideErrorKind::InvalidInput).view(),
        SciOxideErrorView::InvalidInput {
            reason: InvalidInputReason::Empty,
            ..
        }
    ));
    assert_error(
        &aaa(&[0.0_f64, 1.], &[1.]),
        SciOxideErrorKind::DimensionMismatch,
    );
    assert_error(
        &aaa(&[0.0_f64, f64::NAN], &[1., 2.]),
        SciOxideErrorKind::NonFinite,
    );
    assert_error(
        &aaa(&[0.0_f64, 1.], &[1., f64::INFINITY]),
        SciOxideErrorKind::NonFinite,
    );
    assert_error(
        &aaa(&[0.0_f64, 0.], &[1., 2.]),
        SciOxideErrorKind::InvalidInput,
    );
    // The configured form rejects a zero limit and a limit above the sample count.
    for max_terms in [0, 3] {
        assert_error(
            &aaa_with_options(
                &[0.0_f64, 1.],
                &[1., 2.],
                domain(),
                options(max_terms, Tolerance::from_sqrt_epsilon()),
            ),
            SciOxideErrorKind::InvalidInput,
        );
    }
    assert_error(
        &aaa_with_options(
            &[0.0_f64, 1.],
            &[1.],
            domain(),
            AaaOptions::new().with_max_terms(1),
        ),
        SciOxideErrorKind::DimensionMismatch,
    );
}

#[test]
fn default_aaa_options_follow_the_reference_limit_and_sqrt_epsilon() {
    fn check<R: RealScalar>() {
        let options = AaaOptions::<R>::default();
        assert_eq!(options.max_terms(), 100);
        assert_eq!(options.tolerance(), Tolerance::from_sqrt_epsilon());
        assert_eq!(options.tolerance().absolute(), R::SQRT_EPSILON);
        assert_eq!(options.tolerance().relative(), R::SQRT_EPSILON);
        assert_eq!(options.lawson_iterations(), 0);
        assert_eq!(
            options.execution_policy(),
            scioxide_core::ExecutionPolicy::Sequential
        );
        assert_eq!(
            options.svd_options(),
            scioxide_linalg::LinalgExecutionOptions::new()
        );
        assert_eq!(options.budgets(), scioxide_core::SolveBudgets::unbounded());
        assert_eq!(AaaOptions::<R>::new().max_terms(), options.max_terms());
    }
    check::<f32>();
    check::<f64>();
}
